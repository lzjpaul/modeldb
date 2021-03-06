package ai.verta.modeldb.versioning;

import ai.verta.modeldb.entities.versioning.CommitEntity;
import ai.verta.modeldb.exceptions.ModelDBException;
import org.hibernate.Session;

public interface CommitFunction {

  CommitEntity apply(Session session, RepositoryFunction repositoryFunction)
      throws ModelDBException;
}
